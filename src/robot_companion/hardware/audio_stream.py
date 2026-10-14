"""Threaded USB microphone capture with VAD utterance segmentation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from ..models import SpeechUtterance

logger = logging.getLogger(__name__)


class AudioStream:
    """Convert microphone chunks into speech utterances on an asyncio queue."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        chunk_samples: int = 512,
        silence_seconds: float = 0.8,
        max_utterance_seconds: float = 12.0,
        vad: Callable[[tuple[float, ...]], bool] | None = None,
        input_stream: Any | None = None,
        queue_size: int = 4,
        on_speech_start: Callable[[], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self._silence_chunks = max(1, int(silence_seconds * sample_rate / chunk_samples))
        self._max_chunks = max(1, int(max_utterance_seconds * sample_rate / chunk_samples))
        self._vad = vad or self._energy_vad
        self._input_stream = input_stream
        self._on_speech_start = on_speech_start
        self.utterances: asyncio.Queue[SpeechUtterance] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self._stop = threading_event()

    def set_on_speech_start(self, callback: Callable[[], None] | None) -> None:
        self._on_speech_start = callback

    async def start(self) -> None:
        if self._task is not None:
            return
        if self._input_stream is None:
            try:
                import sounddevice as sd  # type: ignore[import-not-found]
            except ImportError:
                logger.warning("sounddevice is unavailable; audio stream will remain idle")
            else:
                self._input_stream = sd.RawInputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.chunk_samples,
                )
                self._input_stream.start()
        self._stop.clear()
        self._task = asyncio.create_task(self._read_loop(), name="audio-capture")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
        if self._input_stream is not None and hasattr(self._input_stream, "stop"):
            self._input_stream.stop()
            self._input_stream.close()

    async def _read_loop(self) -> None:
        speech: list[float] = []
        silent_chunks = 0
        while not self._stop.is_set():
            if self._input_stream is None:
                await asyncio.sleep(0.25)
                continue
            try:
                raw, _ = await asyncio.to_thread(self._input_stream.read, self.chunk_samples)
                chunk = tuple(float(value) for value in raw)
            except (OSError, RuntimeError) as exc:
                logger.error("USB microphone read failed: %s", exc)
                await asyncio.sleep(1.0)
                continue
            is_speech = self._vad(chunk)
            if is_speech:
                if not speech and self._on_speech_start is not None:
                    try:
                        self._on_speech_start()
                    except Exception as exc:
                        logger.error("error in on_speech_start callback: %s", exc)
                speech.extend(chunk)
                silent_chunks = 0
            elif speech:
                silent_chunks += 1
                if silent_chunks >= self._silence_chunks:
                    await self._publish(speech)
                    speech, silent_chunks = [], 0
            if len(speech) >= self._max_chunks * self.chunk_samples:
                await self._publish(speech)
                speech, silent_chunks = [], 0

    async def _publish(self, samples: list[float]) -> None:
        utterance = SpeechUtterance(tuple(samples), self.sample_rate)
        if self.utterances.full():
            _ = self.utterances.get_nowait()
        await self.utterances.put(utterance)

    @staticmethod
    def _energy_vad(chunk: tuple[float, ...], threshold: float = 0.015) -> bool:
        return bool(chunk) and (sum(sample * sample for sample in chunk) / len(chunk)) ** 0.5 > threshold


def threading_event() -> Any:
    """Return an event compatible with ``asyncio.to_thread`` polling."""
    import threading

    return threading.Event()
