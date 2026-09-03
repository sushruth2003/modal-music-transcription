# MuScriptor on Modal

Transcribe music into MIDI and timestamped note events with MuScriptor Large on Modal.

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

Download the pinned model weights once, then deploy the workers:

```bash
uv run modal run -m music_transcription.models::download_model
uv run modal deploy -m music_transcription.pipeline
```

Run the included 30-second synthetic example:

```bash
uv run python -m music_transcription.client submit \
  --audio data/synthetic-chords-30s.wav \
  --wait
```

The command prints a `job_id`. You can omit `--wait`, close the client, and check
or download the durable result later:

```bash
uv run python -m music_transcription.client status --job-id <job_id>
uv run python -m music_transcription.client download --job-id <job_id>
```

Submit up to four files from a directory with:

```bash
uv run python -m music_transcription.client submit-batch \
  --directory local-data \
  --limit 4
```

Supported inputs are WAV, FLAC, MP3, M4A, and OGG. Downloaded results are written
to `outputs/<job_id>/`.

## Architecture

```text
                                      Modal
                         ┌────────────────────────────┐
local client ── upload ─►│ artifact Volume           │
     │                   │ source + generated files  │
     │                   └─────────────┬──────────────┘
     │ spawn JobSpec                   │ paths, not bytes
     ▼                                 ▼
deployed process_job ───────► CPU normalization with ffmpeg
     │                                 │ commit normalized.wav
     │                                 ▼
     └──────────────────────► L4 MuScriptor inference
                                       │ commit MIDI/events/metrics
                                       ▼
                              artifact Volume

job status: submitted → preprocessing → transcribing → completed / failed
                    stored in a persistent Modal Dict
```

The local client uploads audio directly to `music-transcription-artifacts` and
asynchronously invokes the deployed `process_job` function with a small job record.
The CPU container converts the source to mono, 16 kHz PCM and commits it. An L4
container reloads that Volume, loads the pinned MuScriptor checkpoint from the
read-only `music-transcription-models` Volume, and writes the results back.

| Modal primitive | Concrete role |
|---|---|
| Image | Defines the CPU or GPU container filesystem and dependencies |
| Function | Runs one durable CPU coordinator per song |
| GPU class | Keeps one model instance in memory while an L4 container is warm |
| Model Volume | Stores the static checkpoint independently of containers |
| Artifact Volume | Stores source audio, normalized WAV, MIDI, events, and metrics |
| Dict | Stores job state, paths, timestamps, errors, and the result summary |
| `spawn` / `spawn_map` | Submits one job or fans out a batch without waiting locally |

Each job has a stable directory:

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

Containers call `commit()` after writes and `reload()` before reading files written
by another container. The GPU class has `min_containers=0` and `max_containers=4`,
so it scales to zero when idle and processes up to four songs concurrently.
