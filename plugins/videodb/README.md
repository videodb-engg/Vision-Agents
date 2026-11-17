# VideoDB Plugin

This plugin brings [VideoDB](https://videodb.io) scene understanding to Vision
Agents as a `VideoLLM`.

Recent frames from the watched track are buffered and encoded into a short MP4
clip on demand. The clip is uploaded to a VideoDB collection, split into one
time-based scene with several sampled frames, and that scene is described by
VideoDB's vision model using your prompt. Because the model sees frames from
across the clip, it can answer "what just happened?" rather than describing a
single still. The answer is vocalized by the agent's TTS service.

## Installation

```bash
uv add "vision-agents[videodb]"
# or directly
uv add vision-agents-plugins-videodb
```

You can grab an API key at https://console.videodb.io.

## Quick Start

```python
import asyncio
import os
from dotenv import load_dotenv
from vision_agents.core import User, Agent, Runner
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, elevenlabs, videodb
from vision_agents.plugins.getstream import CallSessionParticipantJoinedEvent

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = videodb.VideoDBVLM(
        api_key=os.getenv("VIDEO_DB_API_KEY"),  # or set VIDEO_DB_API_KEY
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My happy AI friend", id="agent"),
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(5)  # let a few seconds of video buffer
            await agent.responses.create("Describe what just happened in the video")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
```

## Configuration

### VideoDBVLM Parameters

- `api_key`: str - VideoDB API key. If not provided, read from the
  `VIDEO_DB_API_KEY` environment variable.
- `collection_id`: str - VideoDB collection clips are uploaded to
  (default: `"default"`).
- `model_name`: str - VideoDB model used to describe the scene, e.g. `"pro"`.
  `None` uses the server default (default: `None`).
- `fps`: float - Frame sampling rate for the buffered clip (default: `1.0`).
- `clip_seconds`: int - Length of the clip analyzed per request (default: `5`).
- `frame_count`: int - Frames VideoDB samples from the clip for the description
  (default: `3`).
- `keep_clips`: bool - Keep uploaded clips in the collection instead of
  deleting them after analysis, so they can be played back or indexed
  in VideoDB later (default: `False`).

## Notes

- Each request uploads a clip, extracts a scene and runs a description, so
  latency is higher than single-frame VLMs. Tune `fps`, `clip_seconds` and
  `frame_count` for your use case.
- VideoDB returns each description whole, so the answer arrives as one chunk
  rather than token by token.
- With `keep_clips=True`, clips accumulate in the collection; delete them from
  the collection when you no longer need them.

## Testing

```bash
# Unit tests (no API key needed)
uv run pytest plugins/videodb/tests -m "not integration"

# Integration tests (needs VIDEO_DB_API_KEY)
export VIDEO_DB_API_KEY="your-key-here"
uv run pytest plugins/videodb/tests -m integration
```

## Links

- [VideoDB Documentation](https://docs.videodb.io/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [GitHub Repository](https://github.com/GetStream/Vision-Agents)
