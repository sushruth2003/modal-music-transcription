# Beat-grid correction

MuScriptor predicts note identity and approximate event times, but its raw event
stream does not by itself provide a reliable musical clock. Beat-grid correction
adds that clock so exported MIDI and sheet music have measured tempo, useful bar
lines, and slightly better onset alignment.

This stage changes rhythmic metadata and timing. It does not change which notes,
pitches, or instruments MuScriptor detects.

## Processing flow

```text
uploaded audio or video
        │
        ▼
FFmpeg: mono 16 kHz normalized.wav
        │ commit to artifact Volume
        ├───────────────────────────────┐
        ▼                               │
BeatGridDetector on CPU                 │
Beat This! final0                       │
        │                               │
        │ BPM, meter, downbeat, beats   │
        ▼                               ▼
MuScriptorTranscriber on L4: decode note events
        │
        ├── measure global onset lag against tracked beats
        ├── write tempo/meter/bar alignment into MIDI
        └── apply the same onset shift to browser events
```

`process_job` remains the coordinator. It first commits `normalized.wav` so the
read-only beat worker can see it. The returned beat-grid payload is recorded in
`preprocessing.json` and passed to the GPU method as a small JSON-safe object;
audio bytes and model objects never cross a remote-call boundary.

## Beat detection and validation

`BeatGridDetector` is a Modal class with 2 CPU cores, 4 GiB of memory, at most
four containers, and a 120-second scale-down window. Its `@modal.enter` hook
loads Beat This! once, so subsequent jobs reaching the same warm container reuse
the detector.

For each recording the detector:

1. loads the normalized mono waveform;
2. predicts beat and downbeat positions with Beat This! `final0`;
3. fits one constant tempo to all detected beats;
4. rejects recordings with fewer than eight beats or a tempo-fit residual above
   5% of one beat period;
5. infers beats per bar only when at least 90% of usable downbeat intervals
   agree; and
6. returns BPM, optional meter, first downbeat, beat positions, elapsed time, and
   an optional fallback reason.

Meter uncertainty does not discard a good tempo. In that case the grid carries
`beats_per_bar: null`, and the MIDI gets tempo information without a potentially
wrong time signature.

The thresholds intentionally match the beat-grid behavior in the pinned
`muscriptor==0.3.0` package.

## How timing is corrected

After MuScriptor finishes decoding, the GPU worker compares all predicted note
onsets with the tracked beat positions. MuScriptor's `BeatGrid.with_onset_delay`
looks for a consistent phase offset across common binary and triplet beat
subdivisions.

The onset shift is accepted only when there are at least 40 distinct onset times,
their phase concentration is at least 0.5, and the measured correction is no more
than 40 ms in either direction. Otherwise the onset correction is `0.0` while the
detected tempo and meter remain usable.

Two related offsets then serve different purposes:

- **Onset delay** is subtracted from every note start and end. It is applied to
  both MIDI generation and `events.jsonl`, keeping source-audio playback, the
  synthesized preview, and the piano roll on the same timeline.
- **MIDI bar offset** is a non-negative structural shift added by MuScriptor's
  MIDI writer so a bar line can land on the first detected downbeat without
  producing negative MIDI ticks. It is deliberately not added to browser events,
  because browser events must remain aligned with the original audio timeline.

## Fallback and failure behavior

No stable grid is normal for very short clips, free-tempo music, rubato
performances, or weak beat detections. Those cases return a `BeatDetectionError`
message as `reason` and continue through MuScriptor without a beat grid. The MIDI
then retains MuScriptor's placeholder tempo behavior.

Only an expected musical detection failure falls back. A missing or malformed
checkpoint, unreadable artifact, dependency error, or worker failure still fails
the job so an infrastructure problem cannot silently degrade every result.

## Checkpoint lifecycle

The model image pins:

```text
muscriptor==0.3.0
beat-this==1.1.0
```

The one-time model materializer also downloads Beat This! `final0` into the model
Volume. The checkpoint is accepted only when both properties match:

```text
size:   81,058,141 bytes
sha256: 8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331
```

It is installed through a temporary file and atomically renamed after
verification. Both the CPU detector and GPU transcriber mount the model Volume
read-only, and the model image runs with Hugging Face offline mode enabled.

After upgrading an existing deployment, rerun materialization before deploying:

```bash
uv run modal run -m music_transcription.models::download_model
uv run modal deploy -m music_transcription.pipeline
```

If MuScriptor is already materialized, the first command keeps those weights and
downloads only the Beat This! checkpoint.

## Metrics and observability

`metrics.json` and the public completed-job response include a `timing` object:

```json
{
  "timing": {
    "beat_grid_detected": true,
    "beat_detection_seconds": 1.23,
    "bpm": 119.98,
    "beats_per_bar": 4,
    "first_downbeat_seconds": 0.51,
    "onset_delay_seconds": 0.018,
    "fallback_reason": null
  }
}
```

The M2 benchmark runner records `beat_detection_seconds` for each completed job.
Beat tracking consumes CPU and may add end-to-end latency, especially on a cold
worker, but it does not occupy the L4 or increase the inference-only GPU cost
estimate. Accuracy and total latency should be evaluated on representative music
before changing the worker size, thresholds, or model.

## Implementation map

| File | Responsibility |
|---|---|
| `music_transcription/config.py` | Package, checkpoint, hash, and worker limits |
| `music_transcription/models.py` | Verified, idempotent checkpoint materialization |
| `music_transcription/beat_grid.py` | Warm CPU detector and steady-tempo validation |
| `music_transcription/preprocess.py` | CPU → beat worker → GPU orchestration |
| `music_transcription/transcribe.py` | Onset measurement, MIDI correction, event correction, metrics |
| `music_transcription/schemas.py` | JSON-safe beat-grid payload types |
| `music_transcription/benchmarks/m2_load.py` | Beat-stage latency capture |
