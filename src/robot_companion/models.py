"""Shared domain models used across hardware, perception, and dialogue."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Mapping


class Language(str, Enum):
    ENGLISH = "en"
    MANDARIN = "zh"


class Emotion(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    CALM = "calm"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class EmotionPosteriors:
    """A normalized probability distribution over supported emotions."""

    values: Mapping[Emotion, float]
    timestamp: float = field(default_factory=monotonic)

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("emotion posterior distribution cannot be empty")
        if any(probability < 0.0 for probability in self.values.values()):
            raise ValueError("emotion probabilities cannot be negative")
        total = sum(self.values.values())
        if total <= 0.0:
            raise ValueError("emotion posterior distribution must have a positive sum")
        object.__setattr__(
            self,
            "values",
            {emotion: probability / total for emotion, probability in self.values.items()},
        )

    @property
    def label(self) -> Emotion:
        return max(self.values, key=self.values.__getitem__)


@dataclass(frozen=True, slots=True)
class SpeechUtterance:
    """A VAD-delimited audio segment."""

    samples: tuple[float, ...]
    sample_rate: int
    language: Language | None = None
    transcript: str | None = None
    timestamp: float = field(default_factory=monotonic)


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """A camera frame with a monotonic capture timestamp."""

    data: object
    timestamp: float = field(default_factory=monotonic)
