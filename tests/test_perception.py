"""Unit tests for perception modules."""

import asyncio
import pytest

from robot_companion.models import Emotion, EmotionPosteriors, Language, SpeechUtterance, VideoFrame
from robot_companion.perception.facial_emotion import FacialEmotionEngine
from robot_companion.perception.late_fusion import LateFusion
from robot_companion.perception.speech_emotion import SpeechEmotionEngine
from robot_companion.perception.speech_recognition import SpeechRecognitionEngine


def test_facial_emotion_default_neutral():
    engine = FacialEmotionEngine()
    frame = VideoFrame(data="dummy")
    result = asyncio.run(engine.predict(frame))
    assert result.label == Emotion.NEUTRAL
    assert result.values[Emotion.NEUTRAL] == 1.0


def test_facial_emotion_mock_prediction():
    # Mock extractor returns array of 4 numbers, classifier returns 7 logits with index 0 (HAPPY) highest
    def mock_extractor(data):
        return [1.0, 2.0]

    def mock_classifier(window_features):
        return [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    engine = FacialEmotionEngine(extractor=mock_extractor, classifier=mock_classifier)
    frame = VideoFrame(data="frame")
    result = asyncio.run(engine.predict(frame))
    assert result.label == Emotion.HAPPY
    assert result.values[Emotion.HAPPY] > 0.9


def test_speech_emotion_default_neutral():
    engine = SpeechEmotionEngine()
    utterance = SpeechUtterance(samples=(0.1, 0.2), sample_rate=16_000)
    result = asyncio.run(engine.predict(utterance))
    assert result.label == Emotion.NEUTRAL
    assert result.values[Emotion.NEUTRAL] == 1.0


def test_speech_emotion_onnx_style_run():
    class MockOnnxSession:
        def run(self, output_names, input_feed):
            # Returns logits with SAD (index 1) highest
            return [[[0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]

    engine = SpeechEmotionEngine(model=MockOnnxSession())
    utterance = SpeechUtterance(samples=(0.1, -0.1), sample_rate=16_000)
    result = asyncio.run(engine.predict(utterance))
    assert result.label == Emotion.SAD
    assert result.values[Emotion.SAD] > 0.9


def test_speech_emotion_callable():
    def mock_model(samples):
        return [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]

    engine = SpeechEmotionEngine(model=mock_model)
    utterance = SpeechUtterance(samples=(0.5,), sample_rate=16_000)
    result = asyncio.run(engine.predict(utterance))
    assert result.label == Emotion.ANGRY


def test_late_fusion_weights():
    with pytest.raises(ValueError, match="non-negative"):
        LateFusion(audio_weight=-1.0, visual_weight=1.0)
    with pytest.raises(ValueError, match="not both zero"):
        LateFusion(audio_weight=0.0, visual_weight=0.0)

    # 50/50 fusion
    fusion = LateFusion(audio_weight=1.0, visual_weight=1.0)
    audio_post = EmotionPosteriors({Emotion.HAPPY: 1.0, Emotion.SAD: 0.0})
    visual_post = EmotionPosteriors({Emotion.HAPPY: 0.0, Emotion.SAD: 1.0})
    fused = fusion.predict(audio_post, visual_post)
    assert fused.values[Emotion.HAPPY] == pytest.approx(0.5)
    assert fused.values[Emotion.SAD] == pytest.approx(0.5)


def test_speech_recognition_fallback():
    engine = SpeechRecognitionEngine()
    utterance = SpeechUtterance(samples=(0.1, 0.2), sample_rate=16_000)
    text, lang = asyncio.run(engine.transcribe(utterance))
    assert text == ""
    assert lang is None


def test_speech_recognition_custom_callable():
    def mock_asr(samples, sr):
        return "hello bear", "en"

    engine = SpeechRecognitionEngine(backend=mock_asr)
    utterance = SpeechUtterance(samples=(0.1, 0.2), sample_rate=16_000)
    text, lang = asyncio.run(engine.transcribe(utterance))
    assert text == "hello bear"
    assert lang == Language.ENGLISH


def test_speech_recognition_language_normalization():
    assert SpeechRecognitionEngine._normalize_language("zh") == Language.MANDARIN
    assert SpeechRecognitionEngine._normalize_language("Chinese") == Language.MANDARIN
    assert SpeechRecognitionEngine._normalize_language("EN-US") == Language.ENGLISH
    assert SpeechRecognitionEngine._normalize_language("es") is None

