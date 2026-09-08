# Alternative architecture evaluation

This isolated experiment reuses the 32 frozen recordings from `QUALITY_SUITE.md`.
It does not import ground-truth notes into remote model workers or change production.

## Models

| Identifier | Checkpoint / architecture | Scope |
| --- | --- | --- |
| `yourmt3_moe` | Official YPTF.MoE+Multi (noPS), Perceiver-TF / experts / multichannel T5 | 32 clips |
| `yourmt3_t5` | Official YMT3+, T5 encoder and single decoder | 32 clips |
| `transkun` | Package 2.0.1, v2 no-pedal-extension, interval transformer / semi-CRF | 8 MAESTRO clips |
| `basic_pitch` | Package 0.4.0, bundled ICASSP 2022 ONNX convolutional model | 32 clips, pitch only |

YourMT3 source and checkpoint files are pinned to the author-hosted Space revision
`5e66c1ea173a8186e0d20432b841d3180cc015b5`. Each result includes checkpoint SHA-256,
installed packages, device and load/inference timing. Separate Python 3.11 images
avoid dependency conflicts. No production checkpoint volume is mounted or modified.
Each model uses one temporary worker, two CPUs, and (except Basic Pitch) one L4.

## Reproduce

First prepare the frozen dataset using `QUALITY_SUITE.md`. Run each model separately:

```bash
uv run modal run --detach -m music_transcription.benchmarks.architecture_modal \
  --model yourmt3_moe --execute
uv run modal run --detach -m music_transcription.benchmarks.architecture_modal \
  --model yourmt3_t5 --execute
uv run modal run --detach -m music_transcription.benchmarks.architecture_modal \
  --model transkun --execute
uv run modal run --detach -m music_transcription.benchmarks.architecture_modal \
  --model basic_pitch --execute
uv run --group eval python -m music_transcription.benchmarks.report_architectures
```

Omit `--execute` to print the selection. Runs save each completed clip and resume
it automatically. `--limit` bounds selection to at most 32 clips. The runner accepts
only <=30-second mono 16 kHz excerpts, verifies their checksums and never uploads
reference MIDI. Image builds are defined for all models, but only the selected worker
performs inference. Timings include resampling, inference and exports, excluding
loading and network transfer. First-call lazy initialization remains in the timing.

## Export adaptation

YourMT3's official demo writer uses `gm_ext_plus`, intentionally combining strings,
brass and other classes for playback. The adapter invokes the same writer twice from
one decoded note list: the untouched demo export and an `MT3_FULL_PLUS` export that
retains fine predicted instruments. Main metrics score the latter; `.demo.mid` files
and separate export metrics remain available. This corrects destructive export
mapping, not predictions. The T5 checkpoint's already-coarse predictions stay coarse.

The first evaluation pass used the demo export. Its 64 MIDI files and run manifests
are retained under `initial-demo-export/`. All 64 repeated demo exports are byte-for-byte
identical, demonstrating that the changed instrument scores came from export handling.
Total executed: 168 passes; final distinct model/recording pairs: 104.

## Evidence boundaries

Primary comparison: eight Slakh Redux test and eight MAESTRO v3 test clips. The
other 16 remain exploratory. YourMT3's published URMP split places six of our eight
ensemble pieces in training; only IDs 01 and 13 are test. GuitarSet membership is
unverified. Official test membership is not an independent checkpoint training audit.

The report suppresses piano-specialist rows outside a fully matched piano subset and
marks Basic Pitch's instrument/drum metrics unavailable. Paired, stratified recording
bootstrap intervals use 5,000 draws and fixed seed. No reference alignment or
threshold tuning is applied. MAESTRO reference notes end at key release, not sustain
pedal release. Input bandwidth is limited to the frozen 16 kHz excerpts for all models.

The report includes the previous raw MuScriptor baseline, app export and beam-4
context, but their timings were collected in the prior run. GPU types agree; precision,
batching and architectures differ. These are usable-system comparisons, not a causal
architecture ablation or an exact end-to-end production latency benchmark.
