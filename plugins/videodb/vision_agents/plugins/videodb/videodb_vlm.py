import asyncio
import io
import logging
import math
import os
import tempfile
import time
import uuid
from collections import deque
from fractions import Fraction
from typing import Any, AsyncIterator, Deque, Optional

import aiortc
import av
import videodb
from videodb import SceneExtractionType
from vision_agents.core import llm
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.utils.video_forwarder import VideoForwarder

logger = logging.getLogger(__name__)


class VideoDBVLM(llm.VideoLLM):
    """Video understanding VLM backed by VideoDB scene descriptions.

    Recent frames from the watched track are buffered and encoded into a short
    MP4 clip on demand. The clip is uploaded to a VideoDB collection, split into
    one time-based scene with several sampled frames, and that scene is
    described by VideoDB's vision model using the supplied prompt.

    Args:
        api_key: VideoDB API key. Falls back to the ``VIDEO_DB_API_KEY``
            environment variable.
        collection_id: VideoDB collection the clips are uploaded to
            (default: ``"default"``).
        model_name: VideoDB model used to describe the scene. ``None`` uses
            the server default.
        fps: Frame sampling rate for the buffered clip (default: 1.0).
        clip_seconds: Length of the clip analyzed per request (default: 5).
        frame_count: Frames VideoDB samples from the clip for the description
            (default: 3).
        keep_clips: Keep uploaded clips in the collection instead of deleting
            them after analysis, so they can be played back or indexed in
            VideoDB later (default: False).
    """

    provider_name = "videodb"

    def __init__(
        self,
        api_key: Optional[str] = None,
        collection_id: str = "default",
        model_name: Optional[str] = None,
        fps: float = 1.0,
        clip_seconds: int = 5,
        frame_count: int = 3,
        keep_clips: bool = False,
    ) -> None:
        super().__init__()

        api_key = api_key or os.getenv("VIDEO_DB_API_KEY")
        if not api_key:
            raise ValueError(
                "api_key is required (pass it directly or set VIDEO_DB_API_KEY)"
            )
        if fps <= 0:
            raise ValueError("fps must be positive")
        if clip_seconds < 1:
            raise ValueError("clip_seconds must be at least 1")
        if frame_count < 1:
            raise ValueError("frame_count must be at least 1")

        self.model = model_name or ""
        self.collection_id = collection_id
        self.fps = fps
        self.clip_seconds = clip_seconds
        self.frame_count = frame_count
        self.keep_clips = keep_clips
        self.client = videodb.connect(api_key=api_key)
        self._collection: Any = None

        self._max_frames = max(1, math.ceil(fps * clip_seconds))
        self._frame_buffer: Deque[av.VideoFrame] = deque(maxlen=self._max_frames)
        self._video_forwarder: Optional[VideoForwarder] = None

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Subscribe to a video track and buffer its frames."""
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping previous one")
            await self.stop_watching_video_track()

        if shared_forwarder is not None:
            # Detach from a previous forwarder so handlers don't accumulate.
            if (
                self._video_forwarder is not None
                and self._video_forwarder is not shared_forwarder
            ):
                await self._video_forwarder.remove_frame_handler(
                    self._on_frame_received
                )

            self._video_forwarder = shared_forwarder
            logger.info("🎥 VideoDB subscribing to shared VideoForwarder")
            self._video_forwarder.add_frame_handler(
                self._on_frame_received,
                fps=self.fps,
                name="videodb_vlm",
            )
        else:
            self._video_forwarder = VideoForwarder(
                input_track=track,  # type: ignore[arg-type]
                max_buffer=self._max_frames,
                fps=self.fps,
                name="videodb_vlm_forwarder",
            )
            self._video_forwarder.add_frame_handler(self._on_frame_received)

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Describe the buffered clip with VideoDB and yield the answer.

        VideoDB returns each scene description whole, so one delta is yielded
        per scene rather than per token.

        Args:
            text: Prompt describing what to analyze in the clip.
            participant: Optional participant the request relates to.
        """
        request_start_time = time.perf_counter()
        inference_id = str(uuid.uuid4())

        frames = list(self._frame_buffer)
        if not frames:
            logger.warning("No frames available, skipping VideoDB analysis")
            yield LLMResponseFinal(original=None, text="")
            return

        first_token_time: Optional[float] = None
        text_chunks: list[str] = []
        sequence_number = 0

        try:
            clip = await asyncio.to_thread(self._encode_clip, frames)
            video = await asyncio.to_thread(self._upload_clip, clip, inference_id)
            try:
                scene_collection = await asyncio.to_thread(
                    video.extract_scenes,
                    extraction_type=SceneExtractionType.time_based,
                    extraction_config={
                        "time": self.clip_seconds,
                        "frame_count": self.frame_count,
                    },
                )
                if scene_collection is None:
                    raise RuntimeError(f"VideoDB returned no scenes: id={video.id}")

                for scene in scene_collection.scenes:
                    description = await asyncio.to_thread(
                        scene.describe,
                        prompt=text,
                        model_name=self.model or None,
                    )
                    if not description:
                        continue

                    chunk = description if not text_chunks else f" {description}"
                    is_first = first_token_time is None
                    ttft_ms: Optional[float] = None
                    if is_first:
                        first_token_time = time.perf_counter()
                        ttft_ms = (first_token_time - request_start_time) * 1000

                    text_chunks.append(chunk)
                    yield LLMResponseDelta(
                        content_index=None,
                        item_id=inference_id,
                        output_index=0,
                        sequence_number=sequence_number,
                        delta=chunk,
                        is_first_chunk=is_first,
                        time_to_first_token_ms=ttft_ms,
                    )
                    sequence_number += 1
            finally:
                if not self.keep_clips:
                    await asyncio.to_thread(self._delete_video, video)

            total_text = "".join(text_chunks)
            latency_ms = (time.perf_counter() - request_start_time) * 1000
            final_ttft_ms: Optional[float] = None
            if first_token_time is not None:
                final_ttft_ms = (first_token_time - request_start_time) * 1000

            self.metrics.on_vlm_inference(
                provider=self.provider_name,
                model=self.model or None,
                latency_ms=latency_ms,
                frames_processed=len(frames),
            )

            logger.debug("VideoDB response received (%d chars)", len(total_text))

            yield LLMResponseFinal(
                original=None,
                text=total_text,
                item_id=inference_id,
                latency_ms=latency_ms,
                time_to_first_token_ms=final_ttft_ms,
                model=self.model or None,
            )

        except Exception as exc:
            logger.exception("Error analyzing clip with VideoDB")
            self.on_llm_error(error=exc)
            yield LLMResponseFinal(original=None, text="")
            return

    async def stop_watching_video_track(self) -> None:
        """Stop video forwarding."""
        if self._video_forwarder is not None:
            # Detach only; a shared forwarder keeps running for other subscribers.
            await self._video_forwarder.remove_frame_handler(self._on_frame_received)
            self._video_forwarder = None
            logger.info("Stopped video forwarding")

    async def _on_frame_received(self, frame: av.VideoFrame) -> None:
        """Buffer the latest sampled frame."""
        self._frame_buffer.append(frame)

    def _upload_clip(self, clip: bytes, name: str) -> Any:
        """Upload an encoded clip to the collection and return the VideoDB video."""
        if self._collection is None:
            self._collection = self.client.get_collection(self.collection_id)

        # The SDK uploads from a file path, so the clip goes through a temp file.
        fd, path = tempfile.mkstemp(suffix=".mp4")
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(clip)
            video = self._collection.upload(file_path=path, name=name)
        finally:
            os.remove(path)
        if video is None:
            raise RuntimeError("VideoDB upload returned no video")
        return video

    def _delete_video(self, video: Any) -> None:
        """Best-effort delete of an uploaded clip; failures are logged, not raised."""
        try:
            video.delete()
        except Exception:
            logger.exception("Failed to delete VideoDB video %s", video.id)

    def _encode_clip(self, frames: list[av.VideoFrame]) -> bytes:
        """Encode buffered frames into an in-memory MP4 clip."""
        first = frames[0]
        # libx264 requires even dimensions.
        width = first.width + first.width % 2
        height = first.height + first.height % 2

        # An exact fractional rate keeps non-integer fps (e.g. 29.97) intact.
        rate = Fraction(self.fps).limit_denominator(1000) or Fraction(1)

        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        try:
            stream = container.add_stream("libx264", rate=rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            frame_time_base = Fraction(rate.denominator, rate.numerator)
            for index, frame in enumerate(frames):
                reformatted = frame.reformat(width=width, height=height, format="rgb24")
                reformatted.pts = index
                reformatted.time_base = frame_time_base
                for packet in stream.encode(reformatted):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        return buffer.getvalue()

    async def close(self) -> None:
        """Stop watching the video track and release resources."""
        await self.stop_watching_video_track()
