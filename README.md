# MuScriptor on Modal

Durable, batch-friendly music transcription using
[MuScriptor Large](https://huggingface.co/MuScriptor/muscriptor-large) on Modal.
Upload an audio file, let a CPU worker normalize it, run transcription on an
NVIDIA L4, and download MIDI plus timestamped note events.

```text
local client
  ├─ uploads source audio ───────────────► artifact Volume
  └─ spawns a small job record ─► CPU normalize ─► L4 transcription
                                      │                    │
                                      └─ job Dict ◄────────┘
                                               │
                                  status / download later
```

Large files move through a persistent Modal Volume rather than function arguments.
Job state lives in a Modal Dict, model weights live in a separate read-only Volume,
and both the CPU and GPU workers scale to zero when idle.

## What you get

Each job produces:

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

The local download command retrieves `events.jsonl`, `transcription.mid`, and
`metrics.json` into `outputs/<job_id>/`.

## Quickstart

### 1. Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A [Modal account](https://modal.com/)
- A free [Hugging Face account](https://huggingface.co/)

Clone the repository and install the small local dependency set:

```bash
git clone https://github.com/sushruth2003/modal-music-transcription.git music-transcription
cd music-transcription
uv sync --dev
uv run modal setup
```

PyTorch, MuScriptor, and `ffmpeg` are installed in remote Modal Images, so they do
not need to be installed on your machine.

### 2. Grant model access

The application code is MIT-licensed, but the MuScriptor weights are gated and
licensed **CC BY-NC 4.0 for non-commercial use**. Accept the terms on the
[MuScriptor Large model page](https://huggingface.co/MuScriptor/muscriptor-large),
then create a read-only Hugging Face token.

Copy the example environment file and replace its placeholder locally:

```bash
cp .env.example .env
```

Create the named Modal Secret used by the one-time downloader:

```bash
uv run modal secret create --from-dotenv .env huggingface-secret
```

`.env` is ignored by Git. The secret is attached only to the model downloader,
not to transcription workers.

### 3. Materialize the model once

Download the pinned MuScriptor Large checkpoint into a persistent Modal Volume:

```bash
uv run modal run -m music_transcription.models::download_model
```

The command validates the checkpoint, writes a readiness marker, and commits the
Volume. Re-running it is safe and skips a complete existing snapshot.

### 4. Deploy the workers

```bash
uv run modal deploy -m music_transcription.pipeline
```

This creates a stable deployed app named `music-transcription`. Repeated client
commands look up that deployment and do not create a new ephemeral app version.

### 5. Transcribe the included fixture

```bash
uv run python -m music_transcription.client submit \
  --audio data/synthetic-chords-30s.wav \
  --wait
```

The command prints a `job_id`. Omitting `--wait` returns after submission; the
remote job continues running.

Check it later and download the result:

```bash
uv run python -m music_transcription.client status --job-id <job_id>
uv run python -m music_transcription.client download --job-id <job_id>
```

To transcribe a directory of up to four supported audio files:

```bash
uv run python -m music_transcription.client submit-batch \
  --directory local-data \
  --limit 4
```

Supported inputs are WAV, FLAC, MP3, M4A, and OGG. Files are normalized to mono,
16 kHz, 16-bit PCM before inference.

## Development

```bash
uv run ruff format --check music_transcription tests
uv run ruff check music_transcription tests
uv run pytest
```

Regenerate the committed synthetic fixture with:

```bash
uv run python scripts/generate_smoke_audio.py
```

The fixture contains generated sine-wave chords rather than third-party music.
Optional CPU-only preflights can validate the two remote Images before using an L4:

```bash
uv run modal run -m music_transcription.verification::verify_audio_image
uv run modal run -m music_transcription.verification::verify_model_image
```

## Cost and scaling

The GPU worker uses NVIDIA L4 instances, scales to zero, and allows at most four
simultaneous containers. The included 30-second fixture is the safest first run.
Actual charges include GPU, CPU, memory, and persistent Volume storage; consult
[Modal's current pricing](https://modal.com/pricing) and your workspace billing
report before submitting a large collection.

The model checkpoint is roughly 5.1 GiB and remains in the model Volume between
runs. Source audio and generated artifacts also remain in their Volume until you
remove them.

## Security and data handling

- No credentials are stored in this repository.
- Never commit `.env`; it is ignored by default.
- Only the downloader receives `HF_TOKEN`.
- Personal audio and generated outputs are ignored by default.
- The included fixture is synthetic and contains no third-party recording.

Before publishing a fork, run a secret scanner and inspect `git diff --cached`.

## License

This repository's code is released under the [MIT License](LICENSE). MuScriptor's
code is also MIT-licensed, while its model weights are separately licensed under
CC BY-NC 4.0. This project does not redistribute those weights; each user downloads
them from Hugging Face after accepting the model terms.
