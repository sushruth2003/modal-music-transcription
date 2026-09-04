# MuScriptor on Modal

Transcribe music into timestamped, instrument-aware notes and MIDI with MuScriptor Large on Modal.

## Quickstart

You need Python 3.12, [`uv`](https://docs.astral.sh/uv/), a
[Modal account](https://modal.com/), and a [Hugging Face account](https://huggingface.co/).

```bash
git clone https://github.com/sushruth2003/modal-music-transcription.git music-transcription
cd music-transcription
uv sync --dev
uv run modal setup
```

Accept the gated, non-commercial CC BY-NC terms on the
[MuScriptor Large model page](https://huggingface.co/MuScriptor/muscriptor-large),
create a read-only Hugging Face token, and put it in a local `.env` file:

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your token.
uv run modal secret create --from-dotenv .env huggingface-secret
```

Materialize the pinned model weights once, then deploy the workers and web app:

```bash
uv run modal run -m music_transcription.models::download_model
uv run modal deploy -m music_transcription.pipeline
```

Modal prints the web URL after deployment. Open it to upload a WAV, FLAC, MP3,
M4A, or OGG file (up to 100 MB), watch the durable job progress, compare the
original audio with a browser-synthesized note preview, inspect the synchronized
piano roll, and download MIDI. The public demo accepts at most three submissions
per IP address per rolling hour and ten submissions globally per UTC day. Audio
longer than ten minutes is rejected before GPU inference.

The CLI remains useful for automation and batches:

```bash
uv run python -m music_transcription.client submit \
  --audio data/synthetic-chords-30s.wav \
  --wait

uv run python -m music_transcription.client submit-batch \
  --directory local-data \
  --limit 4

uv run python -m music_transcription.client status --job-id <job_id>
uv run python -m music_transcription.client download --job-id <job_id>
```

## Architecture

```text
browser / CLI
     │
     │ POST audio                         GET status / audio / events / MIDI
     ▼                                                   ▲
┌────────────────────────────────────────────────────────────────────────┐
│ CPU-only FastAPI ASGI Function                                         │
│ reserve quota → stream upload → direct Volume API → process_job.spawn()│
│ returns 202 + job_id immediately                                       │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ small JobSpec
                               ▼
                    CPU process_job Function
                    ffmpeg → mono 16 kHz WAV
                               │ commit + path reference
                               ▼
                    L4 MuScriptor GPU class
                    pinned 1.4B model loaded once per warm container
                               │ commit
                               ▼
              artifact Volume: source + events + MIDI + metrics
                               │
              job Dict: submitted → preprocessing → transcribing
                                      → completed / failed
```

The web function is an I/O layer, not an inference server. `@modal.asgi_app`
adapts FastAPI to Modal, while `@modal.concurrent` lets one CPU container handle
multiple uploads, polls, and downloads. It uses the Volume client API instead of
mounting the artifact Volume, avoiding filesystem reload conflicts between
concurrent requests. The upload is first bounded and streamed through ephemeral
disk, then committed with immutable request metadata. A persistent rate-limit
Dict counts accepted submissions before FastAPI parses the upload body. One web
container handles up to 25 concurrent requests, keeping the quota update
serialized while status and artifact reads remain concurrent.

`process_job.spawn()` makes submission asynchronous: the HTTP request can end
while a CPU worker normalizes the recording and an L4 worker transcribes it. Those
workers exchange Volume paths rather than audio bytes. The model checkpoint lives
on a separate read-only Volume and is loaded by `@modal.enter` once for each warm
GPU container. GPU inference still has `min_containers=0` and `max_containers=4`,
so it scales to zero and has a bounded spend rate.

The durable artifacts for each job are:

```text
jobs/{job_id}/
├── source.<ext>
├── request.json
├── normalized.wav
├── preprocessing.json
├── events.jsonl
├── transcription.mid
└── metrics.json
```

| Component | Concrete job |
|---|---|
| FastAPI ASGI Function | Accept uploads, return job handles, and serve status/artifacts |
| CPU `process_job` Function | Normalize one recording and coordinate its GPU call |
| L4 GPU class | Keep MuScriptor resident while warm and run inference |
| Model Volume | Persist the static, pinned checkpoint independently of containers |
| Artifact Volume | Persist source audio and generated files across every stage |
| Dict | Hold small status records, timestamps, errors, and result summaries |
| Rate-limit Dict | Enforce rolling per-IP and global daily submission quotas |
| `spawn` / `spawn_map` | Start one durable job or fan out a CLI batch without waiting |

The browser turns paired note-start/note-end events into the piano roll. “Source”
plays the uploaded recording; “Notes” schedules a lightweight Web Audio preview,
so M2 does not need another server-side synthesis worker.
