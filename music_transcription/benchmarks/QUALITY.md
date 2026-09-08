# Audio-to-MIDI quality experiments

See [the initial measured results](QUALITY_RESULTS.md) for the completed URMP comparison.

For the broader four-source evaluation and interactive HTML report, see
[the 32-recording suite](QUALITY_SUITE.md).

These tools compare the actual exported MIDI with a performance-aligned reference.
They are independent of the deployed public application. They reuse the pinned
MuScriptor Large checkpoint read-only on one temporary L4 worker and do not write
to job state, public quotas, or either production artifact path.

## Prepare a reproducible dataset

```bash
uv sync --dev --group eval
uv run --group eval python -m music_transcription.benchmarks.prepare_quality_data
```

This downloads WAV, score MIDI, and note annotations for three preselected URMP
pieces from an immutable public Hugging Face mirror revision. Full source WAVs,
URLs, file sizes, and SHA-256 checksums are retained under the ignored
`data/quality-urmp/` directory. No audio or generated MIDI is committed to Git.

The examples are **08 Spring** (flute/violin), **31 Slavonic** (two trumpets, horn,
trombone), and **40 Miserere** (two flutes, oboe, clarinet, bassoon). The first 30
seconds are resampled to mono 16 kHz PCM16 with no loudness normalization. Notes
with onsets in the inner 25 seconds are evaluated; surrounding audio gives context.

**Do not score against the downloaded `Sco_*.mid` directly.** It is the written
score, not the timing of this performance. The preparer creates `reference.mid`
from URMP's official `Notes_*` annotations: onset seconds, pitch in Hz, duration
seconds. It preserves overlapping unisons on separate MIDI tracks. No ground-truth
timing or pitches enter inference. Only the conditioned variants receive the known
instrument identities from recording metadata.

Primary documentation and attribution:

- [URMP dataset](https://labsites.rochester.edu/air/projects/URMP.html)
- [Annotation format](https://labsites.rochester.edu/air/projects/URMP/URMP_doc.pdf)
- Bochen Li, Xinzhao Liu, Karthik Dinesh, Zhiyao Duan, and Gaurav Sharma,
  “Creating a multi-track classical music performance dataset for multi-modal music
  analysis: Challenges, insights, and applications,” IEEE Transactions on
  Multimedia, 2018.

## Preview and run

```bash
uv run --group eval python -m music_transcription.benchmarks.quality_modal
uv run --group eval modal run -m music_transcription.benchmarks.quality_modal \
  --execute --output outputs/quality-example --max-seconds 900
```

The plain Python preview makes no GPU call. The full matrix has 15 GPU calls:

| Variant | Instrument list | Beam width | Chunk shift |
|---|---|---:|---:|
| baseline | unspecified | 1 | 0 |
| conditioned | known | 1 | 0 |
| beam2 | known | 2 | 0 |
| beam4 | known | 4 | 0 |
| shifted | known | 1 | 2.5 s |

All use the deployed model's fp16 weights, greedy/non-sampling configuration,
CFG coefficient 1, and prelude forcing. The shifted variant prepends 2.5 seconds
of silence and subtracts that delay from returned events; it never drops source
audio. A derived **consensus** variant retains one-to-one agreements within 80 ms
of onset, requiring the same pitch and instrument, and averages their onsets and
offsets. This favors precision at the expense of recall. Cost includes both passes.

For a smaller run, pass `--variants baseline,conditioned,beam2`. The default 900
seconds of measured inference time corresponds to about $0.20 at the historical
L4 rate in project config, excluding model load, startup, idle, CPU, and builds.
This budget is checked **between calls**, not a hard billing limit; a running call
can take up to 300 seconds. The worker scales to zero and the app is ephemeral.
The existing synthetic smoke fixture remains suitable for infrastructure checks;
these real recordings are specifically for the user-requested accuracy experiment.

Partial results and predictions are saved after every successful call. An error
ends the run with `completed: false` and a failure reason. Output folders must be
new so earlier experiments cannot be overwritten accidentally.

## Metrics and report

```bash
uv run --group eval python -m music_transcription.benchmarks.report_quality \
  outputs/quality-example

uv run --group eval python -m music_transcription.benchmarks.quality \
  outputs/quality-example/08_Spring_fl_vn/reference.mid \
  outputs/quality-example/08_Spring_fl_vn/beam2.mid \
  --start 2.5 --end 27.5 --audio-end 30
```

The report command re-exports saved note predictions, scores the exported MIDI,
and writes `REPORT.md`, `per-recording.csv`, and refreshed `results.json`. It never
runs inference. Every example contains a playable source WAV, reference MIDI,
predicted MIDI for each variant, and raw note JSON.

- Pitched onset precision/recall/F1: one-to-one `mir_eval` matches within 50 ms and
  50 cents (pitches are integer MIDI semitones).
- Onset+offset F1: additionally within max(50 ms, 20% reference duration).
- Instrument-aware versions: same MT3_FULL_PLUS instrument group required.
- Frame F1: 10 ms sampled pitch activity, instrument-agnostic, sustaining notes
  included even if their onset preceded the evaluation window.
- Drums: separate onset-only metrics; no drums occur in this initial set.
- Runtime: CUDA synchronized inference, peak allocated GPU memory, model load,
  and inference-only cost estimate. Consensus runtime is the sum of its parents.
- Macro F1 weights recordings equally; micro F1 pools matched, extra, and missed
  note counts. Both are retained in JSON.
- Diagnostic onset-tolerance curves at 25/50/100/200 ms and instrument confusion
  among matched notes help distinguish timing and labeling errors. The headline
  score remains at 50 ms; the evaluator does not optimize a shift against labels.

Predictions are never aligned, snapped, or shifted to maximize reference scores.
Beat-grid correction and MIDI bar offsets are deliberately excluded from all
variants to isolate decoding changes. This is a **decoding evaluation**, not a
complete deployed-app benchmark. Offset scores at the input end are censored to
the same audio endpoint for both reference and prediction.

Three classical ensemble excerpts are an exploratory set. They do not establish
performance on vocals, drums, distorted guitars, studio mixes, or other genres.
MuScriptor training-set overlap is unknown, instrument conditioning uses oracle
metadata, and one timing observation per variant is not a latency benchmark.
Keep thresholds fixed and validate any apparent winner on new recordings before
changing the production default.

## Verification

```bash
uv run --group eval pytest -q
uv run ruff check music_transcription tests scripts
uv run ruff format --check music_transcription tests scripts
```

The scoring tests cover exact matches, duplicate predictions, missing notes,
wrong instruments, timing and duration errors, MIDI round-trips with overlapping
unisons, drum handling, and the one-to-one consensus rule.
