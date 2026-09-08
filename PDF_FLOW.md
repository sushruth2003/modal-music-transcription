# MIDI-to-PDF score flow

The score worker uses **MuseScore Studio 4.7.4**, the official Linux x86-64 release,
with its SHA-256 pinned in the image. MuseScore directly imports polyphonic MIDI,
infers notation/voices and engraves PDF. A private Xvfb display supports the
upstream desktop engine in the CPU container; no GPU or external conversion
service is needed.

MuseScore is the best fit here because it handles both MIDI import and engraving.
Verovio is an excellent notation-rendering library, but does not directly import
MIDI; a separate conversion layer would still be necessary. music21 is useful
for musical analysis and transformations, but delegates PDF engraving to another
engine. No engraving engine can recover every notation choice from performance
MIDI, so PDFs remain automatically quantized drafts.

Sources: [official release](https://github.com/musescore/MuseScore/releases/tag/v4.7.4),
[MuseScore command-line conversion](https://musescore.org/en/print/book/export/html/329750),
[Verovio input formats](https://book.verovio.org/toolkit-reference/input-formats.html),
[music21 converters](https://music21.org/music21docs/moduleReference/moduleConverterSubConverters.html).

## User flow

- Choose **MIDI + score** when uploading audio/video, as before.
- A completed MIDI-only result now offers **Create PDF score**. Rendering uses the
  existing saved MIDI. Playback and MIDI download stay available while it runs.
- The button becomes **View score** when the PDF is ready. Opening the saved result
  resumes status polling. Errors offer a retry without discarding the transcription.
- Empty transcriptions explain why there is no score to render.

`POST /transcriptions/{job_id}/score` returns 202 while pending/rendering and 200
when a PDF already exists. Concurrent repeat requests are coalesced; retries are
limited to three admissions per job. The existing transcription's compute
reservation covers this CPU stage, so it does not consume audio-upload quota.
The job remains `completed`; `score_state` separately reports PDF progress.

PDFs are rendered in a private temporary directory. The worker parses every page
with pypdf, verifies page dimensions/content, then atomically publishes the PDF.
Metadata records engine version, page count, bytes, SHA-256 and rendering time.
A failed render cannot replace an existing PDF with a partial file.

## Repeatable probes

Run the real engraving probe before deployment:

```bash
uv run modal run -m music_transcription.probe_score
```

After deployment, test the deployed renderer:

```bash
uv run modal run -m music_transcription.probe_score --deployed
```

The probe generates deterministic MIDI covering simultaneous chords, independent
bass notes, named ensemble parts, percussion and 128 bars for pagination. It checks
that MusicXML import retains the expected note count, validates PDF structure and
fonts, rasterizes every page, and rejects blank pages or content touching page
boundaries. It also requires malformed MIDI to fail. PNGs support visual inspection;
automated checks cannot prove ideal engraving or the musical accuracy of a transcription.

For the actual HTTP/button workflow:

```bash
uv run python -m music_transcription.probe_pdf_flow
```

This uses the owner's Modal credentials to create a labelled synthetic fixture,
then requests its score through the real HTTP API. It validates the PDF, confirms
repeated requests reuse the result, and verifies the MIDI and model metadata are
unchanged. It runs no transcription and consumes no public audio-upload quota.
Use `--prepare-only` to create a fixture for manual browser testing; then
`--job-id <id>` to verify that same result. Outputs and machine-readable evidence
are saved under `outputs/pdf-probe-20260906/` and are not committed.

Estimated compute per complete probe is below $0.03 (CPU only; not a billing cap).

## Verified release: 6 September 2026

Deployed as Modal **v28** on the existing Auto Transcribe URL. Rollback to the
previous transcription-only migration release is available with
`uv run modal app rollback ap-3xSUG8FZ4gLuWHo4joeGuc v26`.

- **78 tests passed**, plus lint, formatting and whitespace checks.
- The real probe passed before and after deployment: 52/52 polyphonic notes,
  64/64 ensemble/percussion notes in four parts, and 512/512 notes across two PDF
  pages. All four rendered pages were visually inspected; malformed MIDI was
  rejected before launching the renderer.
- The browser's Create PDF score action completed on a synthetic MIDI-only job,
  changed to View score, and survived reload with no console errors.
- The HTTP probe validated the downloaded PDF, its engine version and checksum,
  repeated-request reuse, unchanged MIDI bytes and unchanged model metadata.
  The test score took **3.31 seconds** in the CPU render function (not total cold
  startup latency). Job: `d074f1657f404b7ebcd56820d7002884`.
- Both initial optional-score failures and later score-request failures retain
  the successful transcription and offer a retry. No new GPU inference was
  required for this release's PDF probes.
