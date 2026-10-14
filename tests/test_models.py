"""Unit tests for shared domain models."""

import pytest
from robot_companion.models import Emotion, EmotionPosteriors, Language, SpeechUtterance, VideoFrame


def test_language_enum_values():
    assert Language.ENGLISH.value == "en"
    assert Language.MANDARIN.value == "zh"


def test_emotion_enum_values():
    assert Emotion.HAPPY.value == "happy"
    assert Emotion.NEUTRAL.value == "neutral"
    assert len(Emotion) == 7


def test_emotion_posteriors_normalization():
    posteriors = EmotionPosteriors({
        Emotion.HAPPY: 2.0,
        Emotion.NEUTRAL: 2.0,
    })
    assert posteriors.values[Emotion.HAPPY] == pytest.approx(0.5)
    assert posteriors.values[Emotion.NEUTRAL] == pytest.approx(0.5)
    assert posteriors.label in (Emotion.HAPPY, Emotion.NEUTRAL)


def test_emotion_posteriors_highest_label():
    posteriors = EmotionPosteriors({
        Emotion.HAPPY: 0.1,
        Emotion.SAD: 0.8,
        Emotion.NEUTRAL: 0.1,
    })
    assert posteriors.label == Emotion.SAD


def test_emotion_posteriors_empty_validation():
    with pytest.raises(ValueError, match="cannot be empty"):
        EmotionPosteriors({})


def test_emotion_posteriors_negative_probability():
    with pytest.raises(ValueError, match="cannot be negative"):
        EmotionPosteriors({Emotion.HAPPY: -0.1, Emotion.NEUTRAL: 1.1})


def test_emotion_posteriors_zero_sum():
    with pytest.raises(ValueError, match="positive sum"):
        EmotionPosteriors({Emotion.HAPPY: 0.0, Emotion.NEUTRAL: 0.0})


def test_speech_utterance_defaults():
    utterance = SpeechUtterance(samples=(0.1, 0.2, 0.3), sample_rate=16_000)
    assert utterance.sample_rate == 16_000
    assert utterance.samples == (0.1, 0.2, 0.3)
    assert utterance.language is None
    assert utterance.transcript is None
    assert utterance.timestamp > 0


def test_video_frame_defaults():
    frame = VideoFrame(data="dummy_frame_data")
    assert frame.data == "dummy_frame_data"
    assert frame.timestamp > 0

