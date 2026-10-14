"""Async cloud LLM client for empathetic responses."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..models import Emotion, Language

logger = logging.getLogger(__name__)


class CloudLLMClient:
    """Provider-neutral Gemini/MiMo-compatible JSON client."""

    def __init__(self, endpoint: str, api_key: str, *, timeout: float = 20.0, client: httpx.AsyncClient | None = None) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    async def respond(self, text: str, emotion: Emotion, language: Language) -> str:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "emotion_context": emotion.value,
            "language": language.value,
        }
        close_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.post(self._endpoint, params={"key": self._api_key}, json=payload)
            response.raise_for_status()
            data: Any = response.json()
            return self._extract_text(data)
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            logger.error("cloud dialogue request failed: %s", exc)
            raise RuntimeError("cloud dialogue request failed") from exc
        finally:
            if close_client:
                await client.aclose()

    @staticmethod
    def _extract_text(data: Any) -> str:
        return str(data["candidates"][0]["content"]["parts"][0]["text"])

