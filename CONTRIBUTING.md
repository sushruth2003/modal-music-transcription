# Contributing

Contributions are welcome. Please keep changes small, avoid committing real audio
or generated artifacts, and never include credentials.

Before opening a pull request, run:

```bash
uv sync --dev
uv run ruff format --check music_transcription tests
uv run ruff check music_transcription tests
uv run pytest
```

Changes to the remote runtime should preserve the separation between local client,
CPU preprocessing, GPU inference, model storage, and artifact storage. New GPU
tests should use short synthetic inputs and state their expected cost.
