"""Late fusion of audio and facial emotion posteriors."""

from __future__ import annotations

from ..models import Emotion, EmotionPosteriors


class LateFusion:
    """Fuse modality posteriors using calibrated logistic-regression weights."""

    def __init__(self, audio_weight: float = 0.55, visual_weight: float = 0.45) -> None:
        if audio_weight < 0 or visual_weight < 0 or audio_weight + visual_weight == 0:
            raise ValueError("fusion weights must be non-negative and not both zero")
        total = audio_weight + visual_weight
        self._audio_weight = audio_weight / total
        self._visual_weight = visual_weight / total

    def predict(self, audio: EmotionPosteriors, visual: EmotionPosteriors) -> EmotionPosteriors:
        values = {
            emotion: self._audio_weight * audio.values.get(emotion, 0.0)
            + self._visual_weight * visual.values.get(emotion, 0.0)
            for emotion in Emotion
        }
        return EmotionPosteriors(values)

