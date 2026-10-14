"""Local English/Mandarin translation adapter."""

from __future__ import annotations

import asyncio
from typing import Any

from ..models import Language


class Translator:
    """Wrap an HY-MT-1.8B pipeline without requiring it during import."""

    def __init__(self, pipeline: Any | None = None) -> None:
        self._pipeline = pipeline

    async def translate(self, text: str, source: Language, target: Language) -> str:
        if source is target or not text.strip():
            return text
        if self._pipeline is None:
            raise RuntimeError("translation pipeline is not configured")
        return str(await asyncio.to_thread(self._pipeline, text, source.value, target.value))

