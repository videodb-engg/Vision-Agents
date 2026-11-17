"""
Tests for the VideoDB VideoDBVLM plugin.

Integration tests require the VIDEO_DB_API_KEY environment variable:

    export VIDEO_DB_API_KEY="your-key-here"
    uv run pytest plugins/videodb/tests/test_videodb_vlm.py -m integration -v

To run only unit tests (no API key needed):

    uv run pytest plugins/videodb/tests/test_videodb_vlm.py -m "not integration" -v
"""

import io
import os
from fractions import Fraction

import av
import numpy as np
import pytest

from vision_agents.core.llm.llm import LLMResponseFinal
from vision_agents.plugins.videodb import VideoDBVLM


def _solid_frame(width: int, height: int, pts: int = 0) -> av.VideoFrame:
    """Build a solid-color RGB frame of the given size."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
    frame.pts = pts
    frame.time_base = Fraction(1, 1)
    return frame


class TestVideoDBVLM:
    """Unit tests that do not touch the network."""

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("VIDEO_DB_API_KEY", raising=False)
        with pytest.raises(ValueError):
            VideoDBVLM()

    def test_rejects_non_positive_fps(self):
        with pytest.raises(ValueError):
            VideoDBVLM(api_key="x", fps=0)

    def test_rejects_short_clip(self):
        with pytest.raises(ValueError):
            VideoDBVLM(api_key="x", clip_seconds=0)

    def test_rejects_zero_frame_count(self):
        with pytest.raises(ValueError):
            VideoDBVLM(api_key="x", frame_count=0)

    def test_buffer_size_tracks_fps_and_duration(self):
        vlm = VideoDBVLM(api_key="x", fps=2.0, clip_seconds=5)
        assert vlm._frame_buffer.maxlen == 10

    async def test_empty_buffer_yields_empty_final(self):
        """Without buffered frames nothing is uploaded and the answer is empty."""
        vlm = VideoDBVLM(api_key="x")
        items = [item async for item in vlm.simple_response("What happened?")]
        assert len(items) == 1
        assert isinstance(items[0], LLMResponseFinal)
        assert items[0].text == ""

    def test_encode_clip_is_decodable_with_even_dimensions(self):
        """Odd-sized frames must encode into a decodable MP4 with even dimensions."""
        vlm = VideoDBVLM(api_key="x", fps=1.0, clip_seconds=3)
        frames = [_solid_frame(63, 47, pts=i) for i in range(3)]

        clip = vlm._encode_clip(frames)

        assert isinstance(clip, bytes) and len(clip) > 0
        container = av.open(io.BytesIO(clip))
        try:
            stream = container.streams.video[0]
            assert stream.codec_context.width == 64
            assert stream.codec_context.height == 48
            decoded = [f for f in container.decode(video=0)]
            assert len(decoded) == 3
        finally:
            container.close()


@pytest.mark.integration
class TestVideoDBVLMIntegration:
    async def test_describes_buffered_clip(self):
        """Buffer a few real frames and run a live VideoDB scene description.

        This is an opt-in integration test: it is only collected when the
        ``integration`` marker is selected. If ``VIDEO_DB_API_KEY`` is missing
        the test fails with a clear, actionable error rather than skipping.
        """
        assert os.getenv("VIDEO_DB_API_KEY"), (
            "VIDEO_DB_API_KEY must be set to run the VideoDB integration tests"
        )
        vlm = VideoDBVLM(fps=1.0, clip_seconds=5)
        try:
            for _ in range(5):
                await vlm._on_frame_received(_solid_frame(640, 360))

            items = [
                item async for item in vlm.simple_response("Describe this short video.")
            ]
            final = next(item for item in items if isinstance(item, LLMResponseFinal))
            assert len(final.text.strip()) > 0
        finally:
            await vlm.close()
