"""Application orchestration for the companion robot."""

from __future__ import annotations

import asyncio
import logging
import os

from .dialogue.cloud_llm import CloudLLMClient
from .dialogue.translator import Translator
from .dialogue.tts import TTSEngine
from .hardware.audio_output import AudioOutput
from .hardware.audio_stream import AudioStream
from .hardware.camera_stream import CameraStream
from .models import Emotion, EmotionPosteriors, Language, SpeechUtterance
from .perception.facial_emotion import FacialEmotionEngine
from .perception.late_fusion import LateFusion
from .perception.speech_emotion import SpeechEmotionEngine
from .perception.speech_recognition import SpeechRecognitionEngine

logger = logging.getLogger(__name__)


class CompanionApplication:
    """Run capture and perception continuously while dialogue and speech run independently."""

    def __init__(
        self,
        camera: CameraStream,
        audio: AudioStream,
        speech_emotion: SpeechEmotionEngine,
        facial_emotion: FacialEmotionEngine,
        fusion: LateFusion,
        translator: Translator,
        llm: CloudLLMClient | None = None,
        speech_recognizer: SpeechRecognitionEngine | None = None,
        tts: TTSEngine | None = None,
        audio_output: AudioOutput | None = None,
    ) -> None:
        self.camera, self.audio = camera, audio
        self.speech_emotion, self.facial_emotion = speech_emotion, facial_emotion
        self.fusion, self.translator, self.llm = fusion, translator, llm
        self.speech_recognizer = speech_recognizer or SpeechRecognitionEngine()
        self.tts = tts or TTSEngine()
        self.audio_output = audio_output or AudioOutput()

        # Connect barge-in interruption: when speech onset is detected, halt audio playback immediately
        if self.audio_output is not None:
            self.audio.set_on_speech_start(self.audio_output.interrupt)

        self._stop = asyncio.Event()
        self._latest_visual = EmotionPosteriors(
            {emotion: 1.0 if emotion is Emotion.NEUTRAL else 0.0 for emotion in Emotion}
        )

    async def run(self) -> None:
        await self.camera.start()
        await self.audio.start()
        vision_task = asyncio.create_task(self._vision_loop(), name="vision-perception")
        audio_task = asyncio.create_task(self._audio_loop(), name="audio-perception")
        try:
            await self._stop.wait()
        finally:
            vision_task.cancel()
            audio_task.cancel()
            await asyncio.gather(vision_task, audio_task, return_exceptions=True)
            await self.camera.stop()
            await self.audio.stop()
            if self.audio_output is not None:
                await self.audio_output.stop()

    def stop(self) -> None:
        self._stop.set()

    async def _vision_loop(self) -> None:
        async for frame in self.camera.frames():
            self._latest_visual = await self.facial_emotion.predict(frame)

    async def _audio_loop(self) -> None:
        while not self._stop.is_set():
            utterance = await self.audio.utterances.get()
            speech = await self.speech_emotion.predict(utterance)
            fused = self.fusion.predict(speech, self._latest_visual)

            # Transcribe speech utterance if not already present
            language = utterance.language or Language.ENGLISH
            if not utterance.transcript and self.speech_recognizer is not None:
                transcript, detected_lang = await self.speech_recognizer.transcribe(utterance)
                if detected_lang is not None:
                    language = detected_lang
                if transcript:
                    utterance = SpeechUtterance(
                        samples=utterance.samples,
                        sample_rate=utterance.sample_rate,
                        language=language,
                        transcript=transcript,
                        timestamp=utterance.timestamp,
                    )

            if self.llm is None or not utterance.transcript:
                continue

            try:
                prompt = utterance.transcript
                try:
                    prompt = await self.translator.translate(utterance.transcript, language, language)
                except Exception:
                    prompt = utterance.transcript

                response = await self.llm.respond(
                    prompt,
                    fused.label,
                    language,
                )
            except RuntimeError:
                logger.exception("empathetic dialogue generation failed")
            else:
                logger.info("dialogue response ready: %s", response)
                if self.tts is not None and self.audio_output is not None:
                    try:
                        samples, sr = await self.tts.synthesize(response, language)
                        if samples:
                            await self.audio_output.play(samples, sr)
                    except Exception as exc:
                        logger.error("speech synthesis/playback failed: %s", exc)


def build_application() -> CompanionApplication:
    """Build the default runtime from environment configuration."""
    endpoint = os.getenv("ROBOT_LLM_ENDPOINT")
    api_key = os.getenv("ROBOT_LLM_API_KEY")
    tts_url = os.getenv("ROBOT_TTS_WS_URL")

    llm = CloudLLMClient(endpoint, api_key) if endpoint and api_key else None
    tts = TTSEngine(remote_ws_url=tts_url)
    audio_output = AudioOutput()
    speech_recognizer = SpeechRecognitionEngine()

    return CompanionApplication(
        CameraStream(),
        AudioStream(),
        SpeechEmotionEngine(),
        FacialEmotionEngine(),
        LateFusion(),
        Translator(),
        llm=llm,
        speech_recognizer=speech_recognizer,
        tts=tts,
        audio_output=audio_output,
    )


async def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    application = build_application()
    try:
        await application.run()
    except KeyboardInterrupt:
        application.stop()


if __name__ == "__main__":
    asyncio.run(main())
