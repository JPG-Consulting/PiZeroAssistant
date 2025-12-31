from __future__ import annotations

from assistant.asr.base import ASRProvider
from assistant.asr.stub import StubASRProvider


def create_asr_provider(cfg) -> ASRProvider:
    """
    Create an ASR provider based on configuration.

    Currently supports:
      - stub
    """
    provider = cfg.asr.provider

    if provider == "stub":
        return StubASRProvider(
            return_text=getattr(cfg.asr.stub, "return_text", "<stub-asr>")
        )

    raise ValueError(f"Unknown ASR provider: {provider}")
