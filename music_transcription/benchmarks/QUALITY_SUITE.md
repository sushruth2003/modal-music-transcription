# Broad transcription quality study

See [the completed measured results](QUALITY_SUITE_RESULTS.md) for the recommendation and aggregate scores.

This adds a frozen, balanced four-source suite and interactive HTML report to the
initial three-recording URMP pilot. Production defaults and deployments are unchanged.

## Reproduce

```bash
uv sync --dev --group eval
uv run --group eval python -m music_transcription.benchmarks.prepare_quality_suite
uv run --group eval modal run --detach -m music_transcription.benchmarks.quality_suite_modal \
  --execute --output outputs/quality-suite-new --max-seconds 9000
uv run --group eval python -m music_transcription.benchmarks.report_quality_suite \
  outputs/quality-suite-new
```

Open `index.html` inside the output directory. It works without an external CDN.
Keep its neighboring recording folders to retain source audio and MIDI downloads.
The report includes filters for source, output stage, and quality metric; paired
bootstrap intervals; per-recording regressions; and a piano-roll/listening desk.
Both reference and candidate MIDI use the same neutral browser synthesizer, which
compares pitch, rhythm and duration rather than instrument realism or velocity.

Preparation downloads only selected public files, about 474 MiB for this frozen
selection including the official 56 MB MAESTRO MIDI archive. Audio stays in ignored
`data/` and `outputs/`; no source recordings or generated MIDI are committed.
Selection is written before downloading examples or running the model. The manifest
retains source URLs, immutable mirror revision, archive members, and SHA-256 hashes.
Slakh and MAESTRO use official test splits; training overlap with MuScriptor is unknown.
No user recording or private data is used.

The GPU runner checks bounds, verifies input hashes, and persists each complete clip.
Three L4 workers maximum; read-only checkpoint volume; no public app, artifacts,
quota, or state writes. The 9,000-second budget is about $2 of completed inference
at the project's historical rate, excluding CPU, startup, idle, interrupted work,
and builds. Budget checks occur between clips; three in-flight calls can finish
beyond the budget. Each call has a 1,200-second timeout. This is not a hard billing cap.

For interruption recovery, use the exact same manifest/output with `--resume`.
Completed clips are reused. An incomplete clip's six decoding passes are rerun;
the resumed cost record does not include its earlier partial compute. The first
study run lost its client heartbeat during three Slakh clips; the original record
is retained as `run-before-resume.json`. The resumed run uses detached mode.

## Controlled experiments

Cross beam sizes 1, 2 and 4 with automatic vs known instrument identities: six
inferences per recording. Variant order rotates across recordings to reduce first-call
warm-up bias. Each inference supplies raw notes and the actual MuScriptor app-style
MIDI export, with the same CPU Beat This! detector and grid acceptance rules.

Scoring removes **only** the bar-padding marker written by the MIDI serializer.
It preserves the model's acoustic onset correction, tempo serialization, and note
cleanup. Raw notes are an ablation, not the deployed app's baseline. This evaluates
transcription/MIDI export, not PDF engraving, queue behavior or input-format handling.

The audio-refinement variant is derived locally from automatic greedy, independently
for raw and exported notes. Its fixed spectral-flux rule searches the first three
harmonics, allows an 80 ms maximum onset move, and shifts release by the same amount
except at the audio end. It retains uncertain notes and leaves drums unchanged.
No reference timing, pitches, or tuned-on-suite parameters enter the refinement.

## Interpretation

- 32 recordings: eight each of URMP, Slakh Redux, GuitarSet and MAESTRO v3.
- 926.097 seconds of input audio, 766.097 seconds in the scored inner windows.
- All eight URMP examples are new relative to the first pilot.
- Different GuitarSet lead sheets and eight distinct MAESTRO composer credits reduce
  duplicate-work dependence; broader style/performer correlations remain.
- Primary note matching: one-to-one, 50 ms onset / 50 cents pitch.
- Offset tolerance: max(50 ms, 20% reference duration).
- Instrument scoring uses MuScriptor's supported instrument groups.
- Drum metrics use exact MIDI pitch and only clips with reference drums. False drum
  detections on non-drum clips remain in per-recording JSON rather than that macro.
- Frame score measures instrument-agnostic pitch activity on a 10 ms grid.
- Macro scores weight recordings equally; micro counts are also retained.
- Paired 95% percentile bootstrap: 5,000 recording resamples, stratified by source,
  seed 20260905. No multiple-comparison adjustment or universal-generalization claim.
- Piano and Slakh release labels do not extend through sustain pedal. Audible decay
  and pedal therefore limit interpretation of duration and frame scores.
- Runtime: synchronized model inference; product view adds CPU beats/export;
  refinement adds local CPU overhead, not a same-machine latency benchmark.
- Real singing, live drums, noisy commercial mixes and long recordings remain gaps.

## Checks

```bash
uv run --group eval pytest -q
uv run ruff check music_transcription tests scripts
uv run ruff format --check music_transcription tests scripts
```

Meaningful scoring tests cover note matching, overlap-preserving MIDI serialization,
bar-padding-only normalization, paired resampling, and the bounded audio-only
refinement behavior. Browser checks cover filters, charts, links, responsive layout,
and reference/candidate synthesis.
