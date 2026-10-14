"""Unit tests for dialogue and speech modules."""

import asyncio
import json
import httpx
import pytest

from robot_companion.dialogue.cloud_llm import CloudLLMClient
from robot_companion.dialogue.translator import Translator
from robot_companion.dialogue.tts import TTSEngine
from robot_companion.models import Emotion, Language


def test_cloud_llm_client_success():
    async def _test():
        captured_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            data = {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "I am here for you."}]
                        }
                    }
                ]
            }
            return httpx.Response(200, json=data)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        llm = CloudLLMClient("https://example.com/api", "fake_key", client=client)

        reply = await llm.respond("I feel down", Emotion.SAD, Language.ENGLISH)
        assert reply == "I am here for you."
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert "key=fake_key" in str(req.url)
        body = json.loads(req.content.decode("utf-8"))
        assert body["emotion_context"] == "sad"
        assert body["language"] == "en"

    asyncio.run(_test())


def test_cloud_llm_client_failure():
    async def _test():
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        llm = CloudLLMClient("https://example.com/api", "fake_key", client=client)

        with pytest.raises(RuntimeError, match="cloud dialogue request failed"):
            await llm.respond("Hello", Emotion.NEUTRAL, Language.ENGLISH)

    asyncio.run(_test())


def test_translator_same_language():
    translator = Translator()
    text = asyncio.run(translator.translate("Hello world", Language.ENGLISH, Language.ENGLISH))
    assert text == "Hello world"


def test_translator_pipeline_missing():
    translator = Translator()
    with pytest.raises(RuntimeError, match="not configured"):
        asyncio.run(translator.translate("Hello", Language.ENGLISH, Language.MANDARIN))


def test_translator_with_pipeline():
    def mock_pipeline(text, src, tgt):
        return f"{text}_translated_to_{tgt}"

    translator = Translator(pipeline=mock_pipeline)
    result = asyncio.run(translator.translate("Hello", Language.ENGLISH, Language.MANDARIN))
    assert result == "Hello_translated_to_zh"


def test_tts_engine_empty_text():
    tts = TTSEngine()
    samples, sr = asyncio.run(tts.synthesize(""))
    assert samples == ()
    assert sr == 16_000


def test_tts_engine_fallback_synthesis():
    tts = TTSEngine()
    samples, sr = asyncio.run(tts.synthesize("Hello there my friend"))
    assert len(samples) > 0
    assert sr == 16_000


def test_tts_engine_custom_backend():
    def mock_backend(text, lang):
        return [0.1, 0.2, 0.3], 22_050

    tts = TTSEngine(backend=mock_backend)
    samples, sr = asyncio.run(tts.synthesize("Test", Language.ENGLISH))
    assert samples == (0.1, 0.2, 0.3)
    assert sr == 22_050

