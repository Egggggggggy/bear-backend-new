"""Speech emotion inference with an optional ONNX Runtime/Torch backend."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..models import Emotion, EmotionPosteriors, SpeechUtterance

logger = logging.getLogger(__name__)


class SpeechEmotionEngine:
    """Run XLSR Wav2Vec2 inference off the event loop.

    ``model`` may be an ONNX Runtime session or a callable returning logits.
    The adapter intentionally does not download weights at import time.
    """

    def __init__(self, model: Any | None = None) -> None:
        self._model = model
        self._labels = tuple(Emotion)

    async def predict(self, utterance: SpeechUtterance) -> EmotionPosteriors:
        if self._model is None:
            return self._neutral()
        try:
            logits = await asyncio.to_thread(self._infer, utterance)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("speech emotion inference failed: %s", exc)
            return self._neutral()
        probabilities = self._softmax(tuple(float(item) for item in logits))
        return EmotionPosteriors(dict(zip(self._labels, probabilities, strict=False)))

    def _infer(self, utterance: SpeechUtterance) -> Any:
        if hasattr(self._model, "run"):
            return self._model.run(None, {"input_values": [utterance.samples]})[0][0]
        return self._model(utterance.samples)

    def _neutral(self) -> EmotionPosteriors:
        return EmotionPosteriors({emotion: 1.0 if emotion is Emotion.NEUTRAL else 0.0 for emotion in self._labels})

    @staticmethod
    def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
        import math

        maximum = max(logits)
        exponentials = tuple(math.exp(value - maximum) for value in logits)
        total = sum(exponentials)
        return tuple(value / total for value in exponentials)

