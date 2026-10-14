"""Speech-to-text (ASR) engine with modular backend support."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..models import Language, SpeechUtterance

logger = logging.getLogger(__name__)


class SpeechRecognitionEngine:
    """Transcribe speech utterances off the asyncio event loop.

    Accepts an optional backend (such as Faster-Whisper, HuggingFace pipeline,
    or a custom callable). When unconfigured, safely returns an empty transcript
    or retains any pre-existing transcript on the utterance.
    """

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend

    async def transcribe(self, utterance: SpeechUtterance) -> tuple[str, Language | None]:
        if not utterance.samples:
            return utterance.transcript or "", utterance.language

        if self._backend is None:
            return utterance.transcript or "", utterance.language

        try:
            return await asyncio.to_thread(self._infer, utterance)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("speech recognition inference failed: %s", exc)
            return utterance.transcript or "", utterance.language

    def _infer(self, utterance: SpeechUtterance) -> tuple[str, Language | None]:
        # Support Faster-Whisper style model (has .transcribe)
        if hasattr(self._backend, "transcribe"):
            try:
                import numpy as np

                audio_data = np.array(utterance.samples, dtype=np.float32)
                segments, info = self._backend.transcribe(audio_data, language=None)
                text = " ".join(segment.text.strip() for segment in segments).strip()
                lang = self._normalize_language(getattr(info, "language", None))
                return text, lang
            except Exception as exc:
                logger.error("Faster-Whisper inference failed: %s", exc)
                return "", utterance.language

        # Support callable backend: func(samples, sample_rate) -> str | tuple[str, str | Language]
        if callable(self._backend):
            result = self._backend(utterance.samples, utterance.sample_rate)
            if isinstance(result, tuple) and len(result) >= 2:
                text = str(result[0]).strip()
                lang = self._normalize_language(result[1])
                return text, lang
            return str(result).strip(), utterance.language

        return utterance.transcript or "", utterance.language

    @staticmethod
    def _normalize_language(lang_code: Any) -> Language | None:
        if isinstance(lang_code, Language):
            return lang_code
        if not isinstance(lang_code, str):
            return None
        code = lang_code.lower().strip()
        if code in ("zh", "chinese", "mandarin", "cmn", "zh-cn", "zh-tw"):
            return Language.MANDARIN
        if code in ("en", "english", "en-us", "en-gb"):
            return Language.ENGLISH
        return None

