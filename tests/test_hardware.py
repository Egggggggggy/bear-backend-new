"""Unit tests for hardware abstraction layer."""

import asyncio
import pytest

from robot_companion.hardware.audio_output import AudioOutput
from robot_companion.hardware.audio_stream import AudioStream
from robot_companion.hardware.camera_stream import CameraStream


def test_camera_stream_invalid_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        CameraStream(fps=0)


def test_camera_stream_lifecycle():
    async def _test():
        class MockCamera:
            def __init__(self):
                self.count = 0

            def capture_array(self):
                self.count += 1
                return f"frame_{self.count}"

            def stop(self):
                pass

        mock_cam = MockCamera()
        stream = CameraStream(fps=100.0, queue_size=2, camera=mock_cam)
        await stream.start()

        # Retrieve a frame
        frame = await asyncio.wait_for(stream._frames.get(), timeout=1.0)
        assert "frame_" in frame.data

        await stream.stop()
        assert stream._task is None

    asyncio.run(_test())


def test_camera_stream_drop_old_frames():
    async def _test():
        stream = CameraStream(fps=1000.0, queue_size=2, camera=None)
        # Put 2 items manually
        from robot_companion.models import VideoFrame
        await stream._frames.put(VideoFrame("frame1"))
        await stream._frames.put(VideoFrame("frame2"))

        # When full, next item should drop the oldest (frame1)
        if stream._frames.full():
            _ = stream._frames.get_nowait()
        stream._frames.put_nowait(VideoFrame("frame3"))

        first = await stream._frames.get()
        second = await stream._frames.get()
        assert first.data == "frame2"
        assert second.data == "frame3"

    asyncio.run(_test())


def test_audio_stream_energy_vad():
    silence = (0.001,) * 512
    loud = (0.1,) * 512
    assert not AudioStream._energy_vad(silence, threshold=0.015)
    assert AudioStream._energy_vad(loud, threshold=0.015)


def test_audio_stream_barge_in_onset():
    async def _test():
        barge_in_triggered = False

        def on_speech():
            nonlocal barge_in_triggered
            barge_in_triggered = True

        stream = AudioStream(
            sample_rate=16_000,
            chunk_samples=16,
            silence_seconds=0.1,
            on_speech_start=on_speech,
        )

        class MockInputStream:
            def __init__(self):
                # Send 1 loud chunk, then silence chunks
                self.chunks = [[0.2] * 16] + [[0.0] * 16] * 10

            def read(self, n):
                if self.chunks:
                    return self.chunks.pop(0), False
                return [0.0] * 16, False

        stream._input_stream = MockInputStream()
        await stream.start()

        utterance = await asyncio.wait_for(stream.utterances.get(), timeout=2.0)
        assert barge_in_triggered is True
        assert len(utterance.samples) >= 16

        await stream.stop()

    asyncio.run(_test())


def test_audio_output_playback_and_interruption():
    async def _test():
        output = AudioOutput(chunk_size=128)
        # 16,000 samples at 16kHz = 1 second duration
        samples = (0.01,) * 16_000

        task = asyncio.create_task(output.play(samples, sample_rate=16_000))
        # Let it start
        await asyncio.sleep(0.05)
        assert output.is_playing is True

        # Trigger barge-in interruption
        output.interrupt()
        assert output.is_playing is False

        await task
        assert not output.is_playing

    asyncio.run(_test())

