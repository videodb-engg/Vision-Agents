# VideoDB Example

A small, runnable agent that demonstrates the VideoDB `VideoLLM`.

The agent buffers a few seconds of your webcam, uploads them to VideoDB as a
short clip, and has VideoDB's vision model describe frames sampled from across
the clip, so it can answer "what just happened?". The description is spoken out
loud.

- `videodb_example.py` — VideoDB VLM + Deepgram STT + ElevenLabs TTS, joined to
  a Stream video call.

## Prerequisites

- Python 3.10+
- API keys for:
  - [VideoDB](https://console.videodb.io) — video understanding
  - [Stream](https://getstream.io/try-for-free/) — low-latency video edge
  - [Deepgram](https://deepgram.com/) — speech-to-text
  - [ElevenLabs](https://elevenlabs.io/) — text-to-speech

## Setup

1. From the workspace root, install dependencies:
   ```bash
   uv sync
   ```

2. Create a `.env` file in the workspace root with your keys (see
   [`.env.example`](./.env.example)):
   ```bash
   VIDEO_DB_API_KEY=your_videodb_key
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   DEEPGRAM_API_KEY=your_deepgram_key
   ELEVENLABS_API_KEY=your_elevenlabs_key
   ```

## Running

From the workspace root:

```bash
uv run plugins/videodb/example/videodb_example.py
```

Open the join URL printed in the logs in your browser, turn on your camera, and
wait a few seconds — the agent buffers a short clip, sends it to VideoDB, and
speaks a description of what just happened.

## Tuning

```python
videodb.VideoDBVLM(
    clip_seconds=5,   # length of the analyzed clip
    fps=1.0,          # frame sampling rate for the buffer
    frame_count=3,    # frames VideoDB samples for the description
    keep_clips=False, # keep clips in the collection after analysis
)
```

Each request uploads a clip and runs server-side analysis, so latency is higher
than single-frame VLMs. See the [plugin README](../README.md) for the full
parameter list.

## Learn More

- [VideoDB Documentation](https://docs.videodb.io/)
- [Vision Agents Documentation](https://visionagents.ai/)
