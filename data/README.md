# Test audio

`synthetic-chords-30s.wav` is a deterministic 30-second sequence of generated
sine-wave chords. It contains no sampled or third-party music and is provided
under the repository's MIT License.

Regenerate it from the repository root with:

```bash
uv run python scripts/generate_smoke_audio.py
```
