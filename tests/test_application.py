"""Integration tests for CompanionApplication orchestration."""

import asyncio
import os
import httpx
import pytest

from robot_companion.dialogue.cloud_llm import CloudLLMClient
from robot_companion.dialogue.translator import Translator
from robot_companion.dialogue.tts import TTSEngine
from robot_companion.hardware.audio_output import AudioOutput
from robot_companion.hardware.audio_stream import AudioStream
from robot_companion.hardware.camera_stream import CameraStream
from robot_companion.main import CompanionApplication, build_application
from robot_companion.models import Emotion, Language, SpeechUtterance, VideoFrame
from robot_companion.perception.facial_emotion import FacialEmotionEngine
from robot_companion.perception.late_fusion import LateFusion
from robot_companion.perception.speech_emotion import SpeechEmotionEngine
from robot_companion.perception.speech_recognition import SpeechRecognitionEngine


def test_build_application_with_env(monkeypatch):
    monkeypatch.setenv("ROBOT_LLM_ENDPOINT", "https://api.example.com")
    monkeypatch.setenv("ROBOT_LLM_API_KEY", "test_key")
    monkeypatch.setenv("ROBOT_TTS_WS_URL", "wss://tts.example.com")

    app = build_application()
    assert app.llm is not None
    assert app.speech_recognizer is not None
    assert app.tts is not None
    assert app.audio_output is not None


def test_companion_application_end_to_end():
    async def _test():
        # 1. Mock Camera
        cam = CameraStream(fps=100.0, queue_size=2, camera=None)

        # 2. Mock Audio
        audio = AudioStream(sample_rate=16_000, chunk_samples=16)

        # 3. Perception Engines
        def mock_extractor(data):
            return [1.0]

        def mock_classifier(window):
            # HAPPY is index 0
            return [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        facial_engine = FacialEmotionEngine(extractor=mock_extractor, classifier=mock_classifier)
        speech_engine = SpeechEmotionEngine()
        fusion = LateFusion()

        # 4. ASR Mock
        def mock_asr(samples, sr):
            return "I feel very happy today", "en"

        speech_recognizer = SpeechRecognitionEngine(backend=mock_asr)

        # 5. LLM Mock
        def handler(request: httpx.Request) -> httpx.Response:
            data = {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Glad to hear that!"}]
                        }
                    }
                ]
            }
            return httpx.Response(200, json=data)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        llm = CloudLLMClient("https://example.com/llm", "key", client=client)

        # 6. TTS & AudioOutput Mocks
        played_samples = []

        class MockAudioOutput(AudioOutput):
            async def play(self, samples, sample_rate=16_000):
                played_samples.extend(samples)

        audio_output = MockAudioOutput()
        tts = TTSEngine()
        translator = Translator()

        # Build application
        app = CompanionApplication(
            camera=cam,
            audio=audio,
            speech_emotion=speech_engine,
            facial_emotion=facial_engine,
            fusion=fusion,
            translator=translator,
            llm=llm,
            speech_recognizer=speech_recognizer,
            tts=tts,
            audio_output=audio_output,
        )

        app_task = asyncio.create_task(app.run())

        # Allow start
        await asyncio.sleep(0.05)

        # Push camera frame
        await cam._frames.put(VideoFrame("test_frame"))
        await asyncio.sleep(0.05)
        # Verify vision updated
        assert app._latest_visual.label == Emotion.HAPPY

        # Push audio utterance
        utterance = SpeechUtterance(samples=(0.1,) * 32, sample_rate=16_000)
        await audio.utterances.put(utterance)

        # Wait briefly for audio loop to process
        for _ in range(20):
            if played_samples:
                break
            await asyncio.sleep(0.05)

        assert len(played_samples) > 0

        # Test barge-in wiring
        assert audio._on_speech_start == audio_output.interrupt

        # Clean shutdown
        app.stop()
        await app_task

    asyncio.run(_test())

