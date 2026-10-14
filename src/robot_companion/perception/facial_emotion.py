"""Facial action-unit extraction and temporal emotion classification."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from ..models import Emotion, EmotionPosteriors, VideoFrame

logger = logging.getLogger(__name__)


class FacialEmotionEngine:
    """Extract AU features and classify a short temporal window."""

    def __init__(self, extractor: Any | None = None, classifier: Any | None = None, window: int = 8) -> None:
        self._extractor = extractor
        self._classifier = classifier
        self._features: deque[Any] = deque(maxlen=window)
        self._labels = tuple(Emotion)

    async def predict(self, frame: VideoFrame) -> EmotionPosteriors:
        if self._extractor is None or self._classifier is None:
            return self._neutral()
        try:
            features = await asyncio.to_thread(self._extractor, frame.data)
            self._features.append(features)
            logits = await asyncio.to_thread(self._classifier, tuple(self._features))
            return EmotionPosteriors(dict(zip(self._labels, self._softmax(logits), strict=False)))
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("facial emotion inference failed: %s", exc)
            return self._neutral()

    def _neutral(self) -> EmotionPosteriors:
        return EmotionPosteriors({emotion: 1.0 if emotion is Emotion.NEUTRAL else 0.0 for emotion in self._labels})

    @staticmethod
    def _softmax(logits: Any) -> tuple[float, ...]:
        import math

        values = tuple(float(value) for value in logits)
        maximum = max(values)
        exponentials = tuple(math.exp(value - maximum) for value in values)
        total = sum(exponentials)
        return tuple(value / total for value in exponentials)

