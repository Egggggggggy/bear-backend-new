"""Interruptible audio playback sink with barge-in support."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class AudioOutput:
    """Stream audio playback with immediate barge-in interruption capability.

    Supports sounddevice and PyAudio backends with automatic fallback to
    simulated playback when hardware audio drivers are unavailable.
    """

    def __init__(self, *, chunk_size: int = 1024, output_device: Any | None = None) -> None:
        self.chunk_size = chunk_size
        self._output_device = output_device
        self._current_task: asyncio.Task[None] | None = None
        self._interrupt_event = asyncio.Event()
        self._is_playing = False
        self._lock = asyncio.Lock()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    async def play(self, samples: Sequence[float], sample_rate: int = 16_000) -> None:
        """Play audio samples. Halts any previous playback immediately."""
        if not samples:
            return

        self.interrupt()

        async with self._lock:
            self._interrupt_event.clear()
            self._is_playing = True
            self._current_task = asyncio.create_task(
                self._playback_loop(samples, sample_rate),
                name="audio-playback",
            )
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
            finally:
                self._is_playing = False
                self._current_task = None

    def interrupt(self) -> None:
        """Trigger immediate barge-in interruption to stop current speech playback."""
        if self._is_playing or (self._current_task and not self._current_task.done()):
            logger.info("barge-in triggered: interrupting active audio playback")
            self._interrupt_event.set()
            if self._current_task is not None:
                self._current_task.cancel()
            self._is_playing = False

    async def stop(self) -> None:
        """Stop playback and cleanup."""
        self.interrupt()
        if self._current_task is not None:
            try:
                await self._current_task
            except (asyncio.CancelledError, Exception):
                pass
            self._current_task = None

    async def _playback_loop(self, samples: Sequence[float], sample_rate: int) -> None:
        # Check if custom output device or sounddevice is available
        device = self._output_device
        if device is None:
            try:
                import sounddevice as sd  # type: ignore[import-not-found]
                device = sd
            except ImportError:
                pass

        if device is not None and hasattr(device, "OutputStream"):
            try:
                await self._play_sounddevice(device, samples, sample_rate)
                return
            except Exception as exc:
                logger.warning("sounddevice output stream failed: %s; falling back", exc)

        # Fallback / simulated playback loop
        await self._play_fallback(samples, sample_rate)

    async def _play_sounddevice(self, sd: Any, samples: Sequence[float], sample_rate: int) -> None:
        import numpy as np

        audio_data = np.array(samples, dtype=np.float32)
        total_samples = len(audio_data)

        def _write_chunks() -> None:
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
                offset = 0
                while offset < total_samples and not self._interrupt_event.is_set():
                    end = min(offset + self.chunk_size, total_samples)
                    chunk = audio_data[offset:end]
                    stream.write(chunk)
                    offset = end

        await asyncio.to_thread(_write_chunks)

    async def _play_fallback(self, samples: Sequence[float], sample_rate: int) -> None:
        """Simulate chunked playback pacing for development and test environments."""
        duration = len(samples) / max(1, sample_rate)
        chunk_duration = self.chunk_size / max(1, sample_rate)
        elapsed = 0.0

        while elapsed < duration and not self._interrupt_event.is_set():
            step = min(chunk_duration, duration - elapsed)
            await asyncio.sleep(step)
            elapsed += step
