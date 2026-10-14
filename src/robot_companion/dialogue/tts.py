"""Modular text-to-speech (TTS) engine with local and remote backend support."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any, Sequence

from ..models import Language

logger = logging.getLogger(__name__)


class TTSEngine:
    """Synthesize text into audio samples.

    Supports:
    1. Local synthesis backends (e.g. Piper, Kokoro, or custom callable).
    2. Remote WebSocket TTS services (e.g. Orpheus streaming server).
    3. Safe development fallback synthesis for unit testing and local simulation.
    """

    def __init__(
        self,
        *,
        backend: Any | None = None,
        remote_ws_url: str | None = None,
        voice: str = "tara",
        sample_rate: int = 16_000,
    ) -> None:
        self._backend = backend
        self._remote_ws_url = remote_ws_url
        self.voice = voice
        self.sample_rate = sample_rate

    async def synthesize(
        self, text: str, language: Language = Language.ENGLISH
    ) -> tuple[tuple[float, ...], int]:
        if not text.strip():
            return (), self.sample_rate

        # 1. Custom or local backend
        if self._backend is not None:
            try:
                return await asyncio.to_thread(self._infer_local, text, language)
            except Exception as exc:
                logger.error("local TTS synthesis failed: %s; using fallback", exc)
                return self._fallback_synthesis(text)

        # 2. Remote WebSocket service (e.g. Orpheus)
        if self._remote_ws_url:
            try:
                return await self._synthesize_remote(text)
            except Exception as exc:
                logger.error("remote WebSocket TTS failed: %s; using fallback", exc)
                return self._fallback_synthesis(text)

        # 3. Development fallback
        return self._fallback_synthesis(text)

    def _infer_local(self, text: str, language: Language) -> tuple[tuple[float, ...], int]:
        if callable(self._backend):
            result = self._backend(text, language)
            if isinstance(result, tuple) and len(result) == 2:
                samples, sr = result
                return tuple(float(s) for s in samples), int(sr)
        return self._fallback_synthesis(text)

    async def _synthesize_remote(self, text: str) -> tuple[tuple[float, ...], int]:
        import websockets

        async with websockets.connect(self._remote_ws_url) as ws:  # type: ignore[attr-defined]
            await ws.send(json.dumps({"prompt": text, "voice": self.voice}))
            meta_raw = await ws.recv()
            metadata = json.loads(meta_raw)
            sample_rate = int(metadata.get("sample_rate", self.sample_rate))

            raw_chunks: list[bytes] = []
            while True:
                try:
                    chunk = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    if isinstance(chunk, bytes):
                        raw_chunks.append(chunk)
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):  # type: ignore[attr-defined]
                    break

            import numpy as np

            all_bytes = b"".join(raw_chunks)
            if not all_bytes:
                return self._fallback_synthesis(text)
            audio_arr = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return tuple(float(x) for x in audio_arr), sample_rate

    def _fallback_synthesis(self, text: str) -> tuple[tuple[float, ...], int]:
        """Generate subtle synthetic tone burst proportional to text length for dev testing."""
        duration = min(2.0, max(0.1, len(text.split()) * 0.15))
        total_samples = int(duration * self.sample_rate)
        # 440 Hz gentle sine tone with envelope
        samples = []
        for i in range(total_samples):
            t = i / self.sample_rate
            envelope = math.sin(math.pi * i / total_samples)
            samples.append(0.05 * envelope * math.sin(2 * math.pi * 440 * t))
        return tuple(samples), self.sample_rate

