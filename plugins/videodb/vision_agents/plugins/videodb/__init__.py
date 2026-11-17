"""VideoDB plugin for vision-agents.

Provides VideoDB scene descriptions as a VideoLLM, letting an agent reason
about short video clips from a live track.
"""

from .videodb_vlm import VideoDBVLM
from .videodb_vlm import VideoDBVLM as VLM

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

__all__ = [
    "VideoDBVLM",
    "VLM",
]
