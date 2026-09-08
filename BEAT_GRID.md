# Beat metadata

YourMT3+ MoE predicts notes on the original audio clock. Beat This! adds measured
tempo and optional meter to exported MIDI. It does not shift, quantize, or change
the predicted pitches, durations, or instruments.

## Processing flow

FFmpeg writes mono 16 kHz audio to the artifact Volume. The CPU beat worker reads
it, returns BPM, meter, downbeat positions and a fallback reason, and the L4
transcriber reads the same normalized recording. The pipeline passes Volume paths
and small metadata objects between workers, never the loaded models.

Beat This! runs on 2 CPU cores with 4 GiB requested memory, one maximum container,
and a 30-second idle window. Its pinned final0 checkpoint lives on the read-only
model Volume and loads once per warm worker. The detector keeps the established
MuScriptor 0.3.0 beat-fitting utilities, independently of the new transcription model.

## Validation and export

A usable tempo requires at least eight beats and a constant-tempo fit residual
below 5% of a beat. Meter is included only when at least 90% of usable downbeat
intervals agree. Uncertain meter does not discard an otherwise usable tempo.

The MIDI writer sets tempo at time zero and places an accepted time signature at
the first detected downbeat. The first downbeat is also preserved in job metrics.
Notes retain their absolute times (rounded to MIDI ticks, typically below 0.5 ms).
Different MIDI editors may lay out pickup bars differently. PDF notation remains
an automatically imported draft, not a corrected score.

Without a stable grid, MIDI uses 120 BPM metadata and no explicit time signature.
The notes still retain their timing. Expected musical detection failures fall
back; missing weights, unreadable files and infrastructure failures fail the job.

Both `onset_delay_seconds` and `bar_offset_seconds` are zero for new MoE jobs.
Historical MuScriptor jobs and their stored MIDI/events are unchanged; they may
contain the previous onset correction and MIDI bar padding.

## Checkpoints and deployment

`music_transcription.models::download_yourmt3` validates the exact evaluated MoE
checkpoint and the existing Beat This! final0 bytes/SHA-256 before committing the
model Volume. Inference never downloads weights. Deploy with:

```bash
uv run modal run -m music_transcription.models::download_yourmt3
uv run modal deploy -m music_transcription.pipeline
```
