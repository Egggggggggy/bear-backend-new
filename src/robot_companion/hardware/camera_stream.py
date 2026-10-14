"""Async camera capture with an optional Picamera2 backend."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..models import VideoFrame

logger = logging.getLogger(__name__)


class CameraStream:
    """Capture frames without blocking the asyncio event loop.

    Picamera2 is imported only when ``start`` is called, allowing simulation and
    unit tests to run on non-Jetson development machines.
    """

    def __init__(
        self,
        *,
        fps: float = 15.0,
        size: tuple[int, int] = (1280, 720),
        queue_size: int = 2,
        camera: Any | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._period = 1.0 / fps
        self._size = size
        self._frames: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=queue_size)
        self._camera = camera
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        if self._camera is None:
            try:
                from picamera2 import Picamera2  # type: ignore[import-not-found]
            except ImportError:
                logger.warning("Picamera2 is unavailable; camera stream will remain idle")
            else:
                self._camera = Picamera2()
                config = self._camera.create_video_configuration(
                    main={"size": self._size, "format": "RGB888"}
                )
                self._camera.configure(config)
                self._camera.start()
        self._stop.clear()
        self._task = asyncio.create_task(self._capture_loop(), name="camera-capture")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
        if self._camera is not None and hasattr(self._camera, "stop"):
            self._camera.stop()

    async def frames(self) -> AsyncIterator[VideoFrame]:
        while not self._stop.is_set():
            yield await self._frames.get()

    async def _capture_loop(self) -> None:
        while not self._stop.is_set():
            started = asyncio.get_running_loop().time()
            if self._camera is not None:
                try:
                    frame = await asyncio.to_thread(self._camera.capture_array)
                except (OSError, RuntimeError) as exc:
                    logger.error("CSI camera capture failed: %s", exc)
                    await asyncio.sleep(min(1.0, self._period * 4))
                    continue
                item = VideoFrame(frame)
                if self._frames.full():
                    _ = self._frames.get_nowait()
                self._frames.put_nowait(item)
            await asyncio.sleep(max(0.0, self._period - (asyncio.get_running_loop().time() - started)))

